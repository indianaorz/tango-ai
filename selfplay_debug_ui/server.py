# ── Begin: selfplay_debug_ui/server.py ──
from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, send_from_directory

from .state import DebugState


def start_debug_ui(
    debug_state: DebugState,
    *,
    host: str = "0.0.0.0",
    port: int = 5010,
) -> None:
    """
    Start the debug UI Flask server in a daemon thread.

    Serves:
      GET /                 -> UI
      GET /api/state        -> JSON state + history
      GET /api/image/<p>    -> latest JPEG (single snapshot)
      GET /api/mjpeg/<p>    -> MJPEG stream (continuous)
    """
    base_dir = Path(__file__).resolve().parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(
        "selfplay_debug_ui",
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
        static_url_path="/static",
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def api_state():
        return jsonify(debug_state.to_json())

    @app.get("/api/image/<int:port_id>")
    def api_image(port_id: int):
        jpg = debug_state.get_image_jpg(port_id)
        if not jpg:
            return Response(status=404)
        return Response(jpg, mimetype="image/jpeg")

    @app.get("/api/mjpeg/<int:port_id>")
    def api_mjpeg(port_id: int):
        """
        Browser-friendly multipart MJPEG stream.
        Uses threading Condition to push frames immediately upon generation
        rather than polling via sleep.
        """
        boundary = "frame"

        # Get the condition variable specific to this port
        cond = debug_state.get_render_condition(port_id)

        def gen():
            while True:
                # Wait for notification from the inference loop
                with cond:
                    cond.wait(timeout=1.0) # 1s timeout to keep connection alive if idle
                
                # Fetch fresh data
                jpg = debug_state.get_image_jpg(port_id)
                if jpg:
                    header = (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n"
                        f"\r\n"
                    ).encode("utf-8")
                    yield header
                    yield jpg
                    yield b"\r\n"

        return Response(
            gen(),
            mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(str(static_dir), filename)

    def _run():
        # threaded=True is required for concurrent request handling (MJPEG + state polling)
        app.run(host=host, port=int(port), debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, name="selfplay_debug_ui", daemon=True)
    t.start()
# ── End: selfplay_debug_ui/server.py ──